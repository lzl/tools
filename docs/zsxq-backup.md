# ZSXQ Backup

This workflow backs up the API-visible content of one ZSXQ (知识星球)
group that the current account is already subscribed to. It uses the same
session authorization as the logged-in web client; it does not bypass
membership checks, login, payment, or anti-abuse controls.

## How the method works

The web client exposes a versioned API under `https://api.zsxq.com/v2`:

- `GET /groups/{group_id}` resolves the subscribed group metadata.
- `GET /groups/{group_id}/topics?scope=all&count=30` returns the newest topics.
- The next page is requested with the last topic's `create_time` as
  `end_time`. The workflow deduplicates topic IDs because the API can repeat a
  boundary topic.
- A topic's `talk`, `question`, `task`, or `solution` object contains its text,
  images, files, answer, and any comments included by the API.
- Long-form topics expose an `inline_article_url`. The backup fetches that page,
  extracts its complete article body, removes active markup, and stores both
  `content_html` and searchable `content_text` inside the topic's `article` object.
- `GET /files/{file_id}/download_url` resolves a signed file URL. Image URLs
  are taken from `original`, then `large`, then `thumbnail`.

This is based on the API behavior documented by the open-source projects
[ZSXQ-BackUp](https://github.com/lxzmads/zsxqbackup) and
[RSSHub's 知识星球 route](https://github.com/DIYgod/RSSHub/tree/master/lib/routes/zsxq).
The API is not a stable public developer API, so a response-code or schema
change should be treated as a failed backup rather than silently producing a
partial export.

## Credentials

1. Sign in to `https://wx.zsxq.com` in your own browser and confirm the target
   group is visible to that account.
2. Open browser developer tools and copy the value of the
   `zsxq_access_token` cookie for `wx.zsxq.com`.
3. Put the token in `.env` (keep this file private):

```dotenv
ZSXQ_ACCESS_TOKEN=your_zsxq_access_token
# Optional; use the User-Agent from the same logged-in browser if required.
ZSXQ_USER_AGENT=Mozilla/5.0 ...
```

The token is only sent in the request Cookie header. It is never printed,
written to JSON, or included in Markdown.

## Run the workflow

The group ID is the numeric value in the web URL:

```bash
uv run workflows/backup_zsxq.py GROUP_ID
```

Each run creates a new directory directly under `artifacts/`, matching the
layout used by the repository's other workflows:

```text
group.json       # group metadata
topics.jsonl     # one hydrated topic object per line, including full article HTML/text
topics.md        # full readable text, answers, returned comments, and attachments
media.jsonl      # downloaded attachment manifest
media/           # images and files, grouped by topic ID
zsxq_viewer.html  # offline reader for this exact backup run
zsxq_viewer_data.js # viewer data bundle with artifact-local media paths
summary.json     # counts, paths, and media errors
```

The workflow generates the viewer inside every run directory. It references
only that run's data and uses `.` as its artifact root, so the whole directory
can be moved or archived without breaking topic or attachment links.

Serve the repository root and open the viewer for the run printed by the
workflow:

```bash
python3 -m http.server 8765
# Open: http://localhost:8765/artifacts/RUN_ID/zsxq_viewer.html
```

For a first smoke test, avoid downloading attachments and stop after one page:

```bash
uv run workflows/backup_zsxq.py GROUP_ID \
  --no-media --max-pages 1 --json
```

Useful options:

```bash
# Only digest topics
uv run workflows/backup_zsxq.py GROUP_ID --scope digests

# Keep requests gentle for a large group
uv run workflows/backup_zsxq.py GROUP_ID --delay 2

# Use a stable explicit run directory (must be dedicated to this run)
uv run workflows/backup_zsxq.py GROUP_ID \
  --run-dir artifacts/manual-zsxq-run
```

`--json` keeps stdout as one JSON object; progress is written to stderr. A
successful run with an attachment problem returns
`completed_with_media_errors` and records each failure in `summary.json`.
The topic JSONL and Markdown remain available so the text backup is not lost.

## Run the atom directly

```bash
uv run atoms/backup_zsxq.py GROUP_ID \
  --output-dir artifacts/manual-zsxq-run --json
```

The atom is deliberately self-contained. The workflow invokes it through its
CLI contract, so it can be tested or replaced independently.
